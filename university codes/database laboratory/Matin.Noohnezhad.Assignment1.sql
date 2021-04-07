
--(sub query for question 1) for find that 10 subcategory that wanted in the question.
select top 10 productSubCategory.ProductSubcategoryID, productSubCategory.Name, COUNT(salesOrderDetail.OrderQty) from Production.ProductSubcategory productSubCategory
JOIN Production.Product product on productSubCategory.ProductSubcategoryID = product.ProductSubcategoryID
JOIN Sales.SpecialOfferProduct specialOfferProduct on product.ProductID = specialOfferProduct.ProductID
JOIN Sales.SalesOrderDetail salesOrderDetail on (salesOrderDetail.ProductID = specialOfferProduct.ProductID and salesOrderDetail.SpecialOfferID = specialOfferProduct.SpecialOfferID)
where YEAR(salesOrderDetail.ModifiedDate) = 2003
group by productSubCategory.ProductSubcategoryID, productSubCategory.Name
order by COUNT(salesOrderDetail.OrderQty) desc

--question 1(complete answer)
select  PAddress.City							          [city name],
		productSubCategory.ProductSubcategoryID           [subcategory id],
		productSubCategory.Name                           [subcategory name],
		COUNT(salesOrderDetail.OrderQty)                  [sales quantity] 
			from(select top 10 productSubCategory.ProductSubcategoryID, productSubCategory.Name, COUNT(salesOrderDetail.OrderQty) orderQuantity from Production.ProductSubcategory productSubCategory
					JOIN Production.Product product on productSubCategory.ProductSubcategoryID = product.ProductSubcategoryID
					JOIN Sales.SpecialOfferProduct specialOfferProduct on product.ProductID = specialOfferProduct.ProductID
					JOIN Sales.SalesOrderDetail salesOrderDetail on (salesOrderDetail.ProductID = specialOfferProduct.ProductID and salesOrderDetail.SpecialOfferID = specialOfferProduct.SpecialOfferID)
						where YEAR(salesOrderDetail.ModifiedDate) = 2003
							group by productSubCategory.ProductSubcategoryID, productSubCategory.Name
								order by COUNT(salesOrderDetail.OrderQty) desc) productSubCategory
				JOIN Production.Product product on productSubCategory.ProductSubcategoryID = product.ProductSubcategoryID
				JOIN Sales.SpecialOfferProduct specialOfferProduct on product.ProductID = specialOfferProduct.ProductID
				JOIN Sales.SalesOrderDetail salesOrderDetail on (salesOrderDetail.ProductID = specialOfferProduct.ProductID and salesOrderDetail.SpecialOfferID = specialOfferProduct.SpecialOfferID)
				JOIN Sales.SalesOrderHeader salesOrderHeader on salesOrderHeader.SalesOrderID = salesOrderDetail.SalesOrderID
				JOIN Person.[Address] PAddress on PAddress.AddressID = salesOrderHeader.ShipToAddressID
					group by PAddress.City, productSubCategory.ProductSubcategoryID,productSubCategory.Name,orderQuantity
						order by orderQuantity desc,COUNT(salesOrderDetail.OrderQty) desc

-----------------------------------------------------------------------------------------------------

--(sub query for question 1)finding top 10 products
select top 10 product.ProductID,product.Name productName,count(LineTotal) totalSale from Production.Product product
JOIN Sales.SpecialOfferProduct specialOfferProduct on product.ProductID = specialOfferProduct.ProductID
JOIN Sales.SalesOrderDetail salesOrderDetail on (salesOrderDetail.ProductID = specialOfferProduct.ProductID and salesOrderDetail.SpecialOfferID = specialOfferProduct.SpecialOfferID)
where YEAR(salesOrderDetail.ModifiedDate) = 2003
group by product.ProductID,product.Name
order by totalSale desc

--question 2(complete answer)
select ProductID							,
	   Name						productName ,
	   label = (CASE 
				WHEN ProductID in (select top 10 product.ProductID totalSale from Production.Product product
									JOIN Sales.SpecialOfferProduct specialOfferProduct on product.ProductID = specialOfferProduct.ProductID
									JOIN Sales.SalesOrderDetail salesOrderDetail on (salesOrderDetail.ProductID = specialOfferProduct.ProductID and salesOrderDetail.SpecialOfferID = specialOfferProduct.SpecialOfferID)
									where YEAR(salesOrderDetail.ModifiedDate) = 2003
										group by product.ProductID,product.Name
											order by count(LineTotal) desc) THEN 'top 10 product'
												ELSE 'Normal Sale Product'
													END )	
				from Production.Product product

