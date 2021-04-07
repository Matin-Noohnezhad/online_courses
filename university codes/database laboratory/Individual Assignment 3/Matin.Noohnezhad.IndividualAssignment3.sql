use AdventureWorks
go

-----------------------------------------------------------------
--Question1 solution

CREATE FUNCTION GetProductsSoldToCustomer(
@Year int,
@CustomerID int
)
RETURNS TABLE
AS
RETURN(
select product.ProductID,
	product.Name ProductName,
	product.ProductNumber,
	SUM(SODetail.OrderQty) [Total Quantity]
	 from Sales.Customer customer
	JOIN Sales.SalesOrderHeader SOHeader ON SOHeader.CustomerID = customer.CustomerID
	JOIN Sales.SalesOrderDetail SODetail ON SODetail.SalesOrderID = SOHeader.SalesOrderID
	JOIN Sales.SpecialOfferProduct SOProduct ON (SOProduct.ProductID = SODetail.ProductID and SOProduct.SpecialOfferID = SODetail.SpecialOfferID)
	JOIN Production.Product product ON product.ProductID = SOProduct.ProductID
		WHERE YEAR(SOHeader.OrderDate) = @Year
		AND customer.CustomerID = @CustomerID
		GROUP BY product.ProductID,product.Name,product.ProductNumber
)
GO

select * from GetProductsSoldToCustomer(2002,23)
go

-----------------------------------------------------------------
--Question2 solution
CREATE PROCEDURE PortionOfProductsSoldToCustomer
@Year int,
@CustomerID int
AS
select * into #temp from GetProductsSoldToCustomer(@Year,@CustomerID)

select CustomerAllYearsData.ProductID,
 CustomerAllYearsData.ProductName,
 CustomerAllYearsData.ProductNumber,
 #temp.[Total Quantity] TotalQuantityInSpecifiedYear,
 CustomerAllYearsData.WholeTotalQuantity,
 (CAST(#temp.[Total Quantity] AS FLOAT)/CAST(WholeTotalQuantity AS FLOAT)) SoldPortionInSpecifiedYear
  from (select product.ProductID,
				product.Name ProductName,
				 product.ProductNumber,
				SUM(SODetail.OrderQty) WholeTotalQuantity
				   from Sales.Customer customer
				JOIN Sales.SalesOrderHeader SOHeader ON SOHeader.CustomerID = customer.CustomerID
				JOIN Sales.SalesOrderDetail SODetail ON SODetail.SalesOrderID = SOHeader.SalesOrderID
				JOIN Sales.SpecialOfferProduct SOProduct ON (SOProduct.ProductID = SODetail.ProductID 
				and SOProduct.SpecialOfferID = SODetail.SpecialOfferID)
				JOIN Production.Product product ON product.ProductID = SOProduct.ProductID
				WHERE customer.CustomerID = @CustomerID
				GROUP BY product.ProductID,product.Name,product.ProductNumber) as CustomerAllYearsData
				JOIN #temp ON #temp.ProductName = CustomerAllYearsData.ProductName 
				AND #temp.ProductID = CustomerAllYearsData.ProductID
				AND #temp.ProductNumber = CustomerAllYearsData.ProductNumber

GO


exec PortionOfProductsSoldToCustomer @Year = 2002, @CustomerID = 23

